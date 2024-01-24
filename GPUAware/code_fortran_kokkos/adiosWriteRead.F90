program TestBPWriteReadHeatMap2D
  use mpi
  use adios2

  use, intrinsic :: iso_c_binding
  use, intrinsic :: iso_fortran_env

  use :: flcl_mod
  use :: flcl_util_kokkos_mod
  use :: view_f_mod

  implicit none

  type(adios2_adios) :: adios
  type(adios2_io) :: ioPut, ioGet
  type(adios2_engine) :: bpWriter, bpReader
  type(adios2_variable), dimension(2) :: var_g, var_gIn

  integer(kind=4), dimension(:, :), allocatable :: g, &
                                                   sel_g
  integer(kind=8), dimension(2) :: ishape, istart, icount
  integer(kind=8), dimension(2) :: sel_start, sel_count
  integer :: ierr, irank, isize, step_status
  integer(c_int) :: val
  integer :: in1, in2
  integer :: i1, i2

  integer(kind=4), pointer, dimension(:,:)  :: g_kokkos, gIn_kokkos, gIn_cpu
  type(view_i32_2d_t) :: g_kokkos_v, gIn_kokkos_v, gIn_cpu_v

  call MPI_INIT(ierr)
  call MPI_COMM_RANK(MPI_COMM_WORLD, irank, ierr)
  call MPI_COMM_SIZE(MPI_COMM_WORLD, isize, ierr)

  call kokkos_initialize()
  call kokkos_print_configuration('flcl-config-', 'kokkos.out')
  in1 = 3
  in2 = 4
  icount = (/in1, in2/)
  istart = (/in1*irank, 0/)
  ishape = (/in1*isize, in2/)

  write(*,*)'Rank ',irank,' Writing on Default memory space (one variable) and on Host space (one variable)'
  call kokkos_allocate_view( g_kokkos, g_kokkos_v, 'gpuData', int(in1, c_size_t), int(in2, c_size_t) )
  allocate (g(in1, in2))
  do i2 = 1, in2
    do i1 = 1, in1
      g(i1, i2) = irank + i1
    end do
  end do
!  call random_number(val)
  val=irank
  call init_view(g_kokkos_v, val)

  ! Start adios2 Writer
  call adios2_init(adios, MPI_COMM_WORLD, ierr)
  call adios2_declare_io(ioPut, adios, 'WriteIO', ierr)

  call adios2_define_variable(var_g(1), ioPut, &
                              'bpFloats', adios2_type_integer4, &
                              2, ishape, istart, icount, &
                              adios2_constant_dims, ierr)
  call adios2_define_variable(var_g(2), ioPut, &
                              'bpFloatsGPU', adios2_type_integer4, &
                              2, ishape, istart, icount, &
                              adios2_constant_dims, ierr)

  call adios2_open(bpWriter, ioPut, 'BPFortranKokkos.bp', adios2_mode_write, &
                   ierr)

  call adios2_put(bpWriter, var_g(1), g, ierr)
  call adios2_put(bpWriter, var_g(2), g_kokkos, ierr)

  call adios2_close(bpWriter, ierr)

  if (allocated(g)) deallocate (g)
  call kokkos_deallocate_view( g_kokkos, g_kokkos_v )

  ! Start adios2 Reader in rank 0
  if (irank == 0) then

    write(*,*)'Reading on Default execution space'

    call adios2_declare_io(ioGet, adios, 'ReadIO', ierr)

    call adios2_open(bpReader, ioGet, 'BPFortranKokkos.bp', &
                     adios2_mode_read, MPI_COMM_SELF, ierr)

    call adios2_begin_step(bpReader, adios2_step_mode_read, -1., &
                           step_status, ierr)

    call adios2_inquire_variable(var_gIn(1), ioGet, &
                                 'bpFloats', ierr)
    call adios2_inquire_variable(var_gIn(2), ioGet, &
                                 'bpFloatsGPU', ierr)

    sel_start = (/0, 0/)
    sel_count = (/ishape(1), ishape(2)/)

    call kokkos_allocate_view( gIn_kokkos, gIn_kokkos_v, 'gpuData', int(ishape(1), c_size_t), int(ishape(2), c_size_t) )
    call kokkos_allocate_view( gIn_cpu, gIn_cpu_v, 'cpuData', int(ishape(1), c_size_t), int(ishape(2), c_size_t) )

    call adios2_set_selection(var_gIn(1), 2, sel_start, sel_count, &
                              ierr)
    call adios2_set_selection(var_gIn(2), 2, sel_start, sel_count, &
                              ierr)
    call adios2_get(bpReader, var_gIn(1), gIn_cpu, ierr)
    call adios2_get(bpReader, var_gIn(2), gIn_kokkos, ierr)

    call adios2_end_step(bpReader, ierr)

    call adios2_close(bpReader, ierr)

    write(*,*)'Reading data written on Host'
    call print_view(gIn_cpu_v)
    write(*,*)'Reading data written on the Default memory space'
    call print_view(gIn_kokkos_v)

    if (allocated(sel_g)) deallocate (sel_g)
    call kokkos_deallocate_view( gIn_kokkos, gIn_kokkos_v )

  end if

  call adios2_finalize(adios, ierr)
  call kokkos_finalize()
  call MPI_Finalize(ierr)

end program TestBPWriteReadHeatMap2D
